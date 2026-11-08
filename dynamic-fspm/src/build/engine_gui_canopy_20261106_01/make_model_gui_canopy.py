"""Derive the GUI canopy copy of Engine v2.2 (CONTRACT.md) from an n-plant v2.2 prepared Model.rgg (built by
build/engine_ideotype_20261105_01/prepare_ideo.py with a v22A/v22R profile). Every edit is an exact text replacement asserted to match once.
python make_model_gui_canopy.py <prepared case dir> <out dir> [KEY=VALUE ...]   (KEY = a settings field without SET_, e.g. LAMP_SHAPE=point)
Writes <out>/Model.rgg (+ parametersMicroTom.rgg, leaflet_outline_uv.csv copied) and <out>/gui_canopy_manifest.json.

What changes (labels gui_scenario_tool, engine_v2_2, assumed_lamp_model, relative_light, not_validation):
 1. a SETTINGS block at the top (the user edits numbers there; GroIMP recompiles on save);
 2. init(): settings checked (out of range -> the run refuses with a message), plant positions from the spacing, the lamp from the settings
    (two half panels -> flat / V120 / V90, tilt, height above the substrate, x position), or the prepared lamp kept for LAMP_SHAPE "keep";
 3. simulate(): days from the settings (DAYS + 11 engine intervals, <= the prepared n); at the end of each day the canopy measures
    (leaf area, dry mass, height, trusses, fruits per plant; canopy light; plant-alone light every K days -> neighbour shading) are written to
    <outDir>daily.csv and, in GUI mode, added to the chart datasets; System.exit only headless. Growth code is not touched."""
import hashlib, json, re, shutil, sys
from pathlib import Path

NL = '\r\n'
DEFAULTS = dict(SPACING_CM=10.0, LAMP_HEIGHT_CM=40.0, LAMP_POSITION='"trays"', LAMP_OFFSET_CM=0.0, LAMP_TILT_DEG=0.0, LAMP_SHAPE='"flat"',
                DAYS=60, ISOLATED_EVERY_K=5, SEED=20261020, LAMP_POWER=1200.0, STEM_WIDTH_MULT=1.0, INTERNODE_MULT=1.0)

SETTINGS = '''// ===================== 설정 SETTINGS (숫자만 바꾸고 저장 Ctrl+S → 툴바 simulate) =====================
// 엔진 v2.2 GUI 캐노피 도구. 라벨: gui_scenario_tool, engine_v2_2, assumed_lamp_model, relative_light, not_validation.
// 엔진은 1200 µmol/s 점광원(0.60 m)으로 보정되었습니다. 다른 램프 설정은 시나리오입니다 (C6: 0.45 m에서 건물중 +104 %).
double SET_SPACING_CM = {SPACING_CM};       // 식재 간격 cm, 6–14
double SET_LAMP_HEIGHT_CM = {LAMP_HEIGHT_CM};   // 배지 표면 → 램프 가장 낮은 점, cm, 30–60
String SET_LAMP_POSITION = {LAMP_POSITION}; // "trays" (트레이 중앙 위) | "gaps" (한 간격 옆, x = +p) | "offset" (아래 값 사용)
double SET_LAMP_OFFSET_CM = {LAMP_OFFSET_CM};   // LAMP_POSITION = "offset" 일 때 x 이동 cm, -30–30
double SET_LAMP_TILT_DEG = {LAMP_TILT_DEG};    // 램프 기울기 (긴 축 둘레, 수직에서) 0–45°
String SET_LAMP_SHAPE = {LAMP_SHAPE};    // "flat" | "V120" | "V90" (tray32 바 램프 60 cm, cos^1 발광) | "keep" (준비된 램프 그대로, 점검용)
int SET_DAYS = {DAYS};                   // DAG, 1–60 (엔진 일 = DAG + 11)
int SET_ISOLATED_EVERY_K = {ISOLATED_EVERY_K};       // 혼자일 때 빛 계산 주기 (일), 1–10; 0 = 끔
long SET_SEED = {SEED}L;           // 개체 추첨 시드 (같은 시드 = 같은 개체들)
double SET_LAMP_POWER = {LAMP_POWER};  // 램프 출력 µmol/s (모든 설정에서 같음; 보정 값 1200)
double SET_STEM_WIDTH_MULT = {STEM_WIDTH_MULT};   // 줄기 굵기 배율 (동결 v2.2 = 1.0, 기본 1.5), 0.5–3; 모양만 바뀌고 질량은 그대로
double SET_INTERNODE_MULT = {INTERNODE_MULT};    // 마디 사이 길이 배율 (동결 v2.2 = 1.0), 0.2–2; 기본값 = 기본 설정에서 키 중앙값 150 mm 맞춤
// ==========================================================================================================
'''

BLOCK = r'''
// ===== [GUI CANOPY] BLOCK (build/engine_gui_canopy_20261106_01) =====
boolean GUI_MODE=Boolean.getBoolean("tomato.gui");
String outDir="";
PrintWriter dailyOut=null;
double[] gcAlone=null; int gcAloneDay=-1;
de.grogra.pf.data.DatasetRef gcLeaf=new de.grogra.pf.data.DatasetRef("leaf_area_cm2"),gcDm=new de.grogra.pf.data.DatasetRef("dry_mass_g"),
    gcH=new de.grogra.pf.data.DatasetRef("height_mm"),gcTr=new de.grogra.pf.data.DatasetRef("trusses"),gcFr=new de.grogra.pf.data.DatasetRef("fruits"),
    gcLight=new de.grogra.pf.data.DatasetRef("canopy_light_and_shading");
void gcCheck(String name,double v,double lo,double hi) {
    if(!(v>=lo && v<=hi))throw new IllegalArgumentException("[설정] "+name+" = "+v+" : 허용 범위 "+lo+"–"+hi+" 밖입니다. 맨 위 설정을 고친 뒤 저장하세요.");
}
void gcSettings() {
    gcCheck("SET_SPACING_CM",SET_SPACING_CM,6,14);gcCheck("SET_LAMP_HEIGHT_CM",SET_LAMP_HEIGHT_CM,30,60);gcCheck("SET_LAMP_OFFSET_CM",SET_LAMP_OFFSET_CM,-30,30);
    gcCheck("SET_LAMP_TILT_DEG",SET_LAMP_TILT_DEG,0,45);gcCheck("SET_STEM_WIDTH_MULT",SET_STEM_WIDTH_MULT,0.5,3);gcCheck("SET_INTERNODE_MULT",SET_INTERNODE_MULT,0.2,2);gcCheck("SET_DAYS",SET_DAYS,1,n-11);gcCheck("SET_ISOLATED_EVERY_K",SET_ISOLATED_EVERY_K,0,10);
    if(!(SET_LAMP_SHAPE.equals("flat")||SET_LAMP_SHAPE.equals("V120")||SET_LAMP_SHAPE.equals("V90")||SET_LAMP_SHAPE.equals("keep")||SET_LAMP_SHAPE.equals("flat1")))throw new IllegalArgumentException("[설정] SET_LAMP_SHAPE = "+SET_LAMP_SHAPE+" : flat | V120 | V90 | keep | flat1");
    if(!(SET_LAMP_POSITION.equals("trays")||SET_LAMP_POSITION.equals("gaps")||SET_LAMP_POSITION.equals("offset")))throw new IllegalArgumentException("[설정] SET_LAMP_POSITION = "+SET_LAMP_POSITION+" : trays | gaps | offset");
    rpSeed=SET_SEED;
    double p=SET_SPACING_CM/100.0;int np=plantX.length;
    if(np==32)for(int k=0;k<np;k++){int i=k/8,r=(k%8)/2,c=k%2;plantX[k]=(i-1.5)*(2*p+0.10)+(c-0.5)*p;plantY[k]=(r-1.5)*p;}   // tray32: 4 trays in a row, gap 0.10 m
    else if(np>1)for(int k=0;k<np;k++){int r=k/2,c=k%2;plantX[k]=(c-0.5)*p;plantY[k]=(r-(np/2-1)/2.0)*p;}   // 4 x 2 tray, rows along y (ideotype contract §6)
}
double gcLampX() { double p=SET_SPACING_CM/100.0; if(plantX.length==32)return 0; return SET_LAMP_POSITION.equals("trays")?0:(SET_LAMP_POSITION.equals("gaps")?p:SET_LAMP_OFFSET_CM/100.0); }
// tray32 over gaps (static tray32 v2 CONTRACT §3): full luminaires at the 3 gap centres + one half-output face at each row end, facing inward.
double gcGapX(int g) { return (gcLampCentreX(g)+gcLampCentreX(g+1))/2; }
double gcEdgeVertex(int side) {   // side -1 = left end (face = right half k +1), +1 = right end (face = left half k -1)
    double p=SET_SPACING_CM/100.0,xe=side<0?gcLampCentreX(0)-p-0.05:gcLampCentreX(3)+p+0.05;
    if(SET_LAMP_SHAPE.equals("flat"))return side<0?xe-0.025:xe+0.025;   // flat: 0.05 m half panel centred 0.05 m outside the outline
    return xe;                                                            // V: lowest edge 0.05 m outside the outline, wing rising toward the trays
}
javax.vecmath.Matrix4d gcLampMX(int k,double xv) { javax.vecmath.Matrix4d m=gcLampM(k);m.m03+=xv;return m; }
// One half panel (0.60 m x 0.05 m, emitting along its local +y) of the bar luminaire; k = -1 left, +1 right half.
// flat: both halves face down side by side (= one 0.10 m panel); V120 / V90: each half rotated about the centre line by phi 30 / 45 deg
// (outer edge up, face normal (k sin phi, 0, -cos phi)); then the whole luminaire tilted about its long axis; the lowest point sits
// SET_LAMP_HEIGHT_CM above the substrate surface.
// Number of luminaires: one per tray of 8 (32 plants = tray32 layout, 4 luminaires over the tray centre lines), else one. Each emits SET_LAMP_POWER.
int gcNLamps() { return plantX.length==32?4:1; }
double gcLampCentreX(int i) { double p=SET_SPACING_CM/100.0; return plantX.length==32?(i-1.5)*(2*p+0.10):0.0; }
javax.vecmath.Matrix4d gcLampMAt(int k,int i) { javax.vecmath.Matrix4d m=gcLampM(k);m.m03+=gcLampCentreX(i);return m; }
javax.vecmath.Matrix4d gcLampM(int k) {
    double L=0.60,w=0.05,phi=SET_LAMP_SHAPE.equals("V120")?Math.toRadians(30):(SET_LAMP_SHAPE.equals("V90")?Math.toRadians(45):0),a=Math.toRadians(SET_LAMP_TILT_DEG);
    javax.vecmath.Matrix4d base=new javax.vecmath.Matrix4d(1,0,0,0, 0,0,1,-L/2, 0,-1,0,0, 0,0,0,1);
    javax.vecmath.Matrix4d t=new javax.vecmath.Matrix4d();t.setIdentity();t.setTranslation(new javax.vecmath.Vector3d(k*w/2,0,0));
    javax.vecmath.Matrix4d rv=new javax.vecmath.Matrix4d();rv.rotY(-k*phi);
    javax.vecmath.Matrix4d rt=new javax.vecmath.Matrix4d();rt.rotY(a);
    javax.vecmath.Matrix4d m=new javax.vecmath.Matrix4d(rt);m.mul(rv);m.mul(t);m.mul(base);
    double zmin=0;for(int kk=-1;kk<=1;kk+=2){double xo=kk*w*Math.cos(phi),zo=w*Math.sin(phi);zmin=Math.min(zmin,-xo*Math.sin(a)+zo*Math.cos(a));}
    double z=plantZ[0]+SET_LAMP_HEIGHT_CM/100.0-zmin;
    m.m03+=gcLampX();m.m23+=z;return m;
}
void gcBuildLamp() {
    if(SET_LAMP_SHAPE.equals("keep"))return;
    [ l:Lamp ==> ; ] [ q:Parallelogram, (q.getLight()!=null) ==> ; ] derive();
    if(SET_LAMP_SHAPE.equals("flat1")){   // one full 0.60 x 0.10 m panel (probe / comparison build of the flat lamp)
        [ w:World ==> w [ Null().(setTransform(gcLampM(0))) Parallelogram(0.6f,0.1f).(setLight(new de.grogra.imp3d.objects.AreaLight().(setPower((float)SET_LAMP_POWER),setExponent(0.0f)))) ]; ]
        derive();return;
    }
    if(plantX.length==32 && SET_LAMP_POSITION.equals("offset"))throw new IllegalArgumentException("[설정] 32개체(4 트레이)는 SET_LAMP_POSITION = \"trays\" 또는 \"gaps\" 만 지원합니다.");
    if(plantX.length==32 && SET_LAMP_POSITION.equals("gaps")){
        [ w:World ==> w for(int g=0;g<3;g++)
            ( [ Null().(setTransform(gcLampMX(-1,gcGapX(g)))) Parallelogram(0.6f,0.05f).(setLight(new de.grogra.imp3d.objects.AreaLight().(setPower((float)(SET_LAMP_POWER/2)),setExponent(0.0f)))) ]
              [ Null().(setTransform(gcLampMX(1,gcGapX(g)))) Parallelogram(0.6f,0.05f).(setLight(new de.grogra.imp3d.objects.AreaLight().(setPower((float)(SET_LAMP_POWER/2)),setExponent(0.0f)))) ] )
            [ Null().(setTransform(gcLampMX(1,gcEdgeVertex(-1)))) Parallelogram(0.6f,0.05f).(setLight(new de.grogra.imp3d.objects.AreaLight().(setPower((float)(SET_LAMP_POWER/2)),setExponent(0.0f)))) ]
            [ Null().(setTransform(gcLampMX(-1,gcEdgeVertex(1)))) Parallelogram(0.6f,0.05f).(setLight(new de.grogra.imp3d.objects.AreaLight().(setPower((float)(SET_LAMP_POWER/2)),setExponent(0.0f)))) ]; ]
        derive();return;
    }
    [ w:World ==> w for(int i=0;i<gcNLamps();i++)
        ( [ Null().(setTransform(gcLampMAt(-1,i))) Parallelogram(0.6f,0.05f).(setLight(new de.grogra.imp3d.objects.AreaLight().(setPower((float)(SET_LAMP_POWER/2)),setExponent(0.0f)))) ]
          [ Null().(setTransform(gcLampMAt(1,i))) Parallelogram(0.6f,0.05f).(setLight(new de.grogra.imp3d.objects.AreaLight().(setPower((float)(SET_LAMP_POWER/2)),setExponent(0.0f)))) ] ); ]
    derive();
}
// ----- display colours (2026-11-07, user request): GUI mode only. The light calculation keeps the engine's black organs: gcBlack() puts
// organShader(o) back on every surface right before each lm.compute(); colours are put on after each geometry commit (display only).
// leaflet = green, petiole and rachis = light green; internode and truss axis = brown; flower = yellow; fruit = red.
boolean GC_COLOURS=true;
de.grogra.imp3d.shading.Shader gcGreen=new de.grogra.imp3d.shading.RGBAShader(0.13f,0.55f,0.13f),gcLeafStalk=new de.grogra.imp3d.shading.RGBAShader(0.30f,0.60f,0.15f),
    gcBrown=new de.grogra.imp3d.shading.RGBAShader(0.45f,0.29f,0.10f),gcSideBrown=new de.grogra.imp3d.shading.RGBAShader(0.28f,0.16f,0.05f),gcYellow=new de.grogra.imp3d.shading.RGBAShader(1.0f,0.85f,0.0f),gcRed=new de.grogra.imp3d.shading.RGBAShader(0.85f,0.10f,0.10f);
de.grogra.imp3d.shading.Shader gcColourOf(Organ o) {
    if(o.s1Fruit)return gcRed;
    if(o.kind==9)return gcGreen; if(o.kind==7 || o.kind==8)return gcLeafStalk; if(o.kind==11)return gcYellow; if(o.kind==6 || o.kind==10)return v2IsSide(o.controller)?gcSideBrown:gcBrown;   // side axes darker brown (user 2026-11-07)
    return organShader(o);
}
void gcDisplayColours() { if(!GUI_MODE || !GC_COLOURS)return; for(NativeSurface s:(* NativeSurface *))s.setShader(gcColourOf(org(s.oi))); }
void gcBlack() { for(NativeSurface s:(* NativeSurface *))s.setShader(organShader(org(s.oi))); }
double[] gcLeafAbs() { double[] a=new double[plantIds.length];for(int j=0;j<organs.size();j++){Organ o=org(j);if(o.kind==9)a[o.owner]+=o.photons;}return a; }
// Plant-alone light: the other plants' surfaces are removed, one light pass per plant (same lamp, rays, seed), then all surfaces rebuilt.
// Runs after the day's canopy pass has been read; growth uses only the canopy pass, so the run's states are unchanged.
// The shading of each plant is computed here from the same day's canopy and alone passes and held until the next update
// (check GT1, 2026-11-06: holding the alone light and dividing a later day's canopy light by it gave -400 % while plants grew).
double[] gcShade=null;
void gcAlonePass(double[] canopy) {
    int np=plantIds.length;double[] a=new double[np];
    for(int p=0;p<np;p++) {
        commitGeometry();
        [ s:NativeSurface, (org(s.oi).owner!=p) ==> ; ] derive();
        if(GUI_MODE)gcBlack();lm.setSeed(seed);lm.compute();
        for(NativeSurface s:(* NativeSurface *)){Organ o=org(s.oi);if(o.kind==9 && o.area>0)a[p]+=lm.getAbsorbedPower(s).integrate()*photonPerJoule;}
    }
    commitGeometry();gcAlone=a;gcAloneDay=step;
    gcShade=new double[np];for(int p=0;p<np;p++)gcShade[p]=a[p]>0?1-canopy[p]/a[p]:Double.NaN;
}
double gcTop(int p) {
    double top=plantZ[p];
    for(NativeSurface s:(* NativeSurface *)){if(org(s.oi).owner!=p)continue;float[] v=((de.grogra.imp3d.objects.PolygonMesh)s.getPolygons()).getVertexData();for(int i=2;i<v.length;i+=3)top=Math.max(top,v[i]);}
    return (top-plantZ[p])*1000;
}
double gcMed(double[] v) { double[] c=java.util.Arrays.copyOf(v,v.length);java.util.Arrays.sort(c);int m=c.length;return m%2==1?c[m/2]:(c[m/2-1]+c[m/2])/2; }
double gcMin(double[] v) { double r=v[0];for(double x:v)r=Math.min(r,x);return r; }
double gcMax(double[] v) { double r=v[0];for(double x:v)r=Math.max(r,x);return r; }
void gcStart() {
    dailyOut=new PrintWriter(new FileWriter(outDir+"daily.csv"));
    dailyOut.println("# gui_scenario_tool, engine_v2_2, assumed_lamp_model, relative_light, conditional_on_unvalidated_scale, not_validation");
    dailyOut.println("# settings: spacing_cm="+SET_SPACING_CM+" lamp_height_cm="+SET_LAMP_HEIGHT_CM+" lamp_position="+SET_LAMP_POSITION+" lamp_offset_cm="+SET_LAMP_OFFSET_CM+" lamp_x_m="+gcLampX()+" tilt_deg="+SET_LAMP_TILT_DEG+" shape="+SET_LAMP_SHAPE+" days="+SET_DAYS+" K="+SET_ISOLATED_EVERY_K+" seed="+SET_SEED+" lamp_power="+SET_LAMP_POWER+" plants="+plantIds.length+" model_sla="+SLA_M2_G);
    dailyOut.println("engine_day,dag,plant,leaf_area_cm2,dry_mass_g,height_mm,trusses,fruits,absorbed_canopy_umol_s,absorbed_alone_umol_s,alone_day,neighbour_shading");
    if(GUI_MODE){
        gcLeaf.clear().setColumnKey(0,"median").setColumnKey(1,"min").setColumnKey(2,"max");gcDm.clear().setColumnKey(0,"median").setColumnKey(1,"min").setColumnKey(2,"max");
        gcH.clear().setColumnKey(0,"median").setColumnKey(1,"min").setColumnKey(2,"max");gcTr.clear().setColumnKey(0,"median").setColumnKey(1,"min").setColumnKey(2,"max");
        gcFr.clear().setColumnKey(0,"median").setColumnKey(1,"min").setColumnKey(2,"max");
        gcLight.clear().setColumnKey(0,"canopy light %").setColumnKey(1,"mean neighbour shading %").setColumnKey(2,"index = light / shading");
        int XY=de.grogra.pf.ui.ChartPanel.XY_PLOT;chart(gcLeaf,XY);chart(gcDm,XY);chart(gcH,XY);chart(gcTr,XY);chart(gcFr,XY);chart(gcLight,XY);
    }
}
// End of each engine day (after growth and geometry). DAG = engine day - 11.07.
// Called right after the day's canopy light pass (same geometry), before growth.
void gcMaybeAlone(double[] canopy) { if(SET_ISOLATED_EVERY_K>0 && plantIds.length>1 && step>=11 && (gcAloneDay<0 || step-gcAloneDay>=SET_ISOLATED_EVERY_K))gcAlonePass(canopy); }
void gcDay(double[] canopy) {
    int np=plantIds.length;double dag=step-11.07;
    double[] la=new double[np],dm=new double[np],h=new double[np],tr=new double[np],fr=new double[np],sh=new double[np];double tot=0,shm=0;int nsh=0;
    for(int j=0;j<organs.size();j++){Organ o=org(j);int p=o.owner;if(o.kind==9)la[p]+=o.area*1e4;dm[p]+=o.mass;if(o.kind==10)tr[p]+=1;if(o.s1Fruit)fr[p]+=1;}
    for(int p=0;p<np;p++){h[p]=gcTop(p);tot+=canopy[p];
        double al=gcAlone==null?Double.NaN:gcAlone[p];sh[p]=gcShade==null?Double.NaN:gcShade[p];if(!Double.isNaN(sh[p])){shm+=sh[p];nsh++;}
        dailyOut.println(step+","+dag+","+plantIds[p]+","+la[p]+","+dm[p]+","+h[p]+","+tr[p]+","+fr[p]+","+canopy[p]+","+al+","+gcAloneDay+","+sh[p]);}
    double light=100*tot/(SET_LAMP_POWER*gcNLamps()),meanSh=nsh>0?100*shm/nsh:Double.NaN,index=(nsh>0 && meanSh>=0.5)?light/meanSh:Double.NaN;
    dailyOut.println(step+","+dag+",CANOPY,"+gcMed(la)+","+gcMed(dm)+","+gcMed(h)+","+gcMed(tr)+","+gcMed(fr)+","+tot+",,"+gcAloneDay+","+(nsh>0?shm/nsh:Double.NaN)+",light_pct="+light+",index="+(Double.isNaN(index)?"—":(""+index)));
    dailyOut.flush();
    if(GUI_MODE && dag>=0){   // one row per day (x = row number = DAG); columns median / min / max
        gcRow(gcLeaf,gcMed(la),gcMin(la),gcMax(la));gcRow(gcDm,gcMed(dm),gcMin(dm),gcMax(dm));gcRow(gcH,gcMed(h),gcMin(h),gcMax(h));
        gcRow(gcTr,gcMed(tr),gcMin(tr),gcMax(tr));gcRow(gcFr,gcMed(fr),gcMin(fr),gcMax(fr));
        gcRow(gcLight,light,nsh>0?meanSh:Double.NaN,index);
    }
}
void gcRow(de.grogra.pf.data.DatasetRef d,double a,double b,double c) { d.addRow().set(0,Double.valueOf(a)).set(1,Double.valueOf(b)).set(2,Double.valueOf(c)); }
'''


def main():
    case, out = Path(sys.argv[1]), Path(sys.argv[2]); kv = dict(a.split('=', 1) for a in sys.argv[3:])
    gui_home = kv.pop('GUI_HOME', '').replace('\\', '/')   # absolute folder of the delivered GUI (outline library + gui_runs); '' for test cases
    W = case / 'native' / 'working'; src = (W / 'Model.rgg').read_bytes().decode('utf-8'); t = src.replace('\r\n', '\n')

    def rep(old, new):
        nonlocal t
        assert t.count(old) == 1, ('gui canopy patch point', t.count(old), old[:90]); t = t.replace(old, new)
    s = dict(DEFAULTS); s.update({k: (f'"{v}"' if k in ('LAMP_POSITION', 'LAMP_SHAPE') else v) for k, v in kv.items()})
    imports = [m.end() for m in re.finditer(r'^import [^\n]*\n', t, flags=re.M)]
    i = imports[-1] if imports else 0
    t = t[:i] + SETTINGS.format(**s) + t[i:]
    # init(): settings, positions and lamp
    rep('    initialize();if(observationMode)', '    gcSettings();\n    initialize();if(observationMode)')
    # display colours (user, 2026-11-07): black before every light pass, colours after every geometry commit — GUI mode only
    n_lm = t.count('lm.setSeed(seed);lm.compute();'); assert n_lm == 2, ('light pass points in the model', n_lm)
    t = t.replace('lm.setSeed(seed);lm.compute();', 'if(GUI_MODE)gcBlack();lm.setSeed(seed);lm.compute();')
    rep('setShader(organShader(o)))]; ]\n    derive();\n}', 'setShader(organShader(o)))]; ]\n    derive();\n    gcDisplayColours();\n}')
    # Amendment 4 (user, 2026-11-07): stem width and internode length multipliers as settings (GUI only; 1.0 = frozen v2.2)
    rep('o.wid=STEM_WIDTH_M*o.ioWidScale*fraction;', 'o.wid=STEM_WIDTH_M*SET_STEM_WIDTH_MULT*o.ioWidScale*fraction;')
    t, nsub = re.subn(r'(double v2InterScaleOf\(int controller,int rank\) \{ return [0-9.]+)\*', r'\1*SET_INTERNODE_MULT*', t)
    assert nsub == 1, ('internode scale patch point', nsub)
    rep('    println("VTCDEV_READY "+runId);\n}', '    gcBuildLamp();\n    println("VTCDEV_READY "+runId);\n}')
    # simulate(): output folder, days, canopy measures, exit only headless
    for name in ('events', 'states', 'flux', 'ledger', 'meshes', 'consumed', 'apices', 'edges', 'proposals', 'axis_flux', 'random_draws'):
        t = t.replace(f'new FileWriter("{name}.csv")', f'new FileWriter(outDir+"{name}.csv")')
    rep('    loadOutlineLibrary();attachGraph();', '    gcStart();\n    loadOutlineLibrary();attachGraph();')
    rep('    for(apply(n)) {\n', '    for(apply(SET_DAYS+11)) {\n')
    rep('        for(int p=0;p<plantIds.length;p++) {\n            if(lightOnly)', '        double[] gcCanopy=gcLeafAbs();gcMaybeAlone(gcCanopy);\n        for(int p=0;p<plantIds.length;p++) {\n            if(lightOnly)')
    rep('        exportState(step+1);checkpoint(step+1);println("VTCDEV_INTERVAL "+(++step));\n',
        '        exportState(step+1);checkpoint(step+1);println("VTCDEV_INTERVAL "+(++step));\n        gcDay(gcCanopy);\n')
    rep('    events.close();states.close();', '    dailyOut.println("# done engine_day="+step);dailyOut.close();\n    events.close();states.close();')
    rep('println("VTCDEV_DONE "+runId);System.exit(0);\n}', 'println("VTCDEV_DONE "+runId);if(!GUI_MODE)System.exit(0);\n}')
    # lamp-only sensor grid (empty scene after the run) also kept under the GUI's own name (the bridge removes its outputs when it refuses a run)
    rep('    sensors.close();\n    } else {',
        '    sensors.close();Files.copy(Paths.get(outDir+"sensors.csv"),Paths.get(outDir+"gui_sensors.csv"),StandardCopyOption.REPLACE_EXISTING);\n'
        '    if(GUI_MODE){[ s:Sensor ==> ; ] derive();commitGeometry();println("[GUI] 실행 끝: 빛 점검용 센서를 치우고 마지막 날 식물을 다시 그렸습니다.");}   // 2026-11-07: the empty-scene sensor pass had left the view without plants\n'
        '    } else {')
    t = t.replace('new FileWriter("sensors.csv")', 'new FileWriter(outDir+"sensors.csv")')
    # GUI mode: every run into its own folder gui_runs/<yyyyMMdd_HHmmss>/ (headless: outDir "" -> the frozen file names)
    rep('public void simulate() {\n',
        'public void simulate() {\n    if(GUI_MODE){outDir="gui_runs/"+new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new java.util.Date())+"/";'
        'Files.createDirectories(Paths.get(outDir));println("[GUI] 결과 폴더: "+outDir);}\n')
    rep('new FileWriter("completion.json")', 'new FileWriter(outDir+"completion.json")')
    rep('done.print("\\""+names[i]+"\\":\\""+hash(names[i])+"\\"");', 'done.print("\\""+names[i]+"\\":\\""+hash(outDir+names[i])+"\\"");')
    # 2026-11-07 (user error "FileNotFoundException: events.csv (액세스가 거부되었습니다)"): GroIMP started outside START_GUI_CANOPY.bat has its
    # install folder (read-only) as working folder. GUI mode is then switched on automatically, outputs go to an absolute run folder and the
    # outline library is read by absolute path. Headless bridge runs (writable working folder, no tomato.gui flag) are unchanged.
    rep('new FileReader("leaflet_outline_uv.csv")', 'new FileReader(gcIn("leaflet_outline_uv.csv"))')
    rep('    if(GUI_MODE){outDir="gui_runs/"+new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new java.util.Date())+"/";',
        '    if(GUI_MODE){outDir=gcRunBase()+new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new java.util.Date())+"/";')
    t = t.rstrip('\n') + '\n' + BLOCK.replace(
        'boolean GUI_MODE=Boolean.getBoolean("tomato.gui");',
        f'String GC_HOME="{gui_home}";\n'
        'boolean GUI_MODE=Boolean.getBoolean("tomato.gui")||!Files.isWritable(Paths.get(System.getProperty("user.dir")));\n'
        'String gcIn(String name) { return (GUI_MODE && GC_HOME.length()>0 && Files.exists(Paths.get(GC_HOME,name)))?GC_HOME+"/"+name:name; }\n'
        'String gcRunBase() {\n'
        '    if(GC_HOME.length()>0 && Files.isWritable(Paths.get(GC_HOME)))return GC_HOME+"/gui_runs/";\n'
        '    if(Files.isWritable(Paths.get(System.getProperty("user.dir"))))return "gui_runs/";\n'
        '    return System.getProperty("user.home").replace(\'\\\\\',\'/\')+"/TomatoCanopy_gui_runs/";\n'
        '}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'Model.rgg').write_bytes(t.replace('\n', NL).encode('utf-8'))
    for nme in ('parametersMicroTom.rgg', 'leaflet_outline_uv.csv'):
        if (W / nme).exists(): shutil.copy2(W / nme, out / nme)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    json.dump(dict(source_case=str(case), source_model_sha256=sha(W / 'Model.rgg'), gui_model_sha256=sha(out / 'Model.rgg'), settings=s), open(out / 'gui_canopy_manifest.json', 'w'), indent=1)
    print(out / 'Model.rgg', sha(out / 'Model.rgg')[:16])


if __name__ == '__main__':
    main()
